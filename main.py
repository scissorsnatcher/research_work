# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from pandas import Series
from os import walk, path
import time

#--- Globals
N = 1750
src_folder_name = 'ПортКавказ'
f_number = 20    
P = 1/19
P_scale = 3
p_threshold = 0.65
bordr_min = 5
bordr_max = 15
start_pres_key = 13 # Enter
close_wins_key = 27 # Esc
#---

directory = path.dirname(path.abspath(__file__))
folder = path.join(directory, src_folder_name)
print(folder)
all_files = next(walk(folder), (None, None, []))[2]
files = all_files[:f_number]
print(f'''Num files for mean: {f_number}
Num files for presentation: {len(all_files)-f_number}
Start reading files...''')

start_time = time.time()
images = []
for fname in files:
    kadr = np.zeros((4096+1750,4096),np.ubyte)
    with open(folder+'/'+fname, 'rb') as file:
        z = file.read(64)
        for i in range(4096):
            z0 = list(file.read(8))
            z1 = list(file.read(4096))
            kadr[i] = z1
            if i < N:
                kadr[i+4096] = z1
    images.append(kadr.copy())
mean_kadr = np.mean(([k for k in images]), axis=0)

kadrs = Series(dtype=object)
for fname in all_files[f_number:]:
    kadr = np.zeros((4096+1750,4096),np.ubyte)
    with open(folder+'/'+fname, 'rb') as file:
        z = file.read(64)
        for i in range(4096):
            z0 = list(file.read(8))
            z1 = list(file.read(4096))
            kadr[i] = z1
            if i < N:
                kadr[i+4096] = z1
    kadrs[fname] = kadr.copy()
    
fin_time = time.time() - start_time
print(f'Reading finish: {int(fin_time//60)} min {round(fin_time%60,4)} s')

class RectMouse:
    
    def __init__(self, image: np.ndarray, color:int=250, width:int=8, main_win_scale:float=8):
        self._src_img = image.copy()
        self.temp_src = image.copy()        
        self.temp_roi = image.copy()
    
        self.PREV, self.CURR = (100,3650), (350,4450)
        self.WINDOW_NAME = "Display"
        self._roi_winname = "rectROI"
        self._binary_winname = "binROI"
        self._roi_img = None
        self.rect_color = color
        self.rect_width = width
        self.scale = main_win_scale
        self.presentation = False
        self.files_query = all_files[f_number:].copy()
        self.exp_dir = {'up':False, 'down':False, 'right':False, 'left':False}
        self.narr_dir = {'up':False, 'down':False, 'right':False, 'left':False}
        self.periph_mean = np.mean([0,255])
    
    def draw_rectangle(self, frame: np.ndarray, pt1: tuple, pt2: tuple):
        self.temp_src = cv.rectangle(frame.copy(), pt1, pt2, self.rect_color, self.rect_width)
        
    def restore_img(self):
        self.temp_src = self._src_img.copy()
        self.temp_roi = self._src_img.copy()
        
    def stop_presentation(self):
        self.presentation = False
        self.files_query = all_files[f_number:].copy()
        
    def contrast_img(self, image: np.ndarray):
        v_min = image.min()
        v_max = image.max()
        contr_func = np.vectorize(lambda x: round((((x-v_min)/(v_max-v_min))**0.5)*255))
        return contr_func(image.copy())
    
    def get_rect_coords(self):
        minX = self.PREV[0] if (self.PREV[0] < self.CURR[0]) else self.CURR[0];
        minY = self.PREV[1] if (self.PREV[1] < self.CURR[1]) else self.CURR[1];
        maxX = self.CURR[0] if (self.PREV[0] < self.CURR[0]) else self.PREV[0];
        maxY = self.CURR[1] if (self.PREV[1] < self.CURR[1]) else self.PREV[1];
        width = maxX - minX;
        height = maxY - minY;
        return minY, minX, height, width
    
    def get_rect_kadr(self, kadr: np.ndarray):
        minX = self.PREV[0] if (self.PREV[0] < self.CURR[0]) else self.CURR[0];
        minY = self.PREV[1] if (self.PREV[1] < self.CURR[1]) else self.CURR[1];
        maxX = self.CURR[0] if (self.PREV[0] < self.CURR[0]) else self.PREV[0];
        maxY = self.CURR[1] if (self.PREV[1] < self.CURR[1]) else self.PREV[1];
        width = maxX - minX;
        height = maxY - minY;
        if width <= 0 or height <= 0:
            return None
        return kadr[minY:minY+height, minX:minX+width]
    
    def check_border_exp(self, frame):
        self.exp_dir = {'up':False, 'down':False, 'right':False, 'left':False}
        Y_len, X_len = frame.shape
        # верхняя граница
        for i in range(bordr_min):
            if frame[i,:].sum() > 0:
                self.exp_dir['up'] = True
                break
        # нижняя граница
        for i in range(Y_len-1,Y_len-bordr_min-1,-1):
            if frame[i,:].sum() > 0:
                self.exp_dir['down'] = True
                break
        # левая граница
        for j in range(bordr_min):
            if frame[:,j].sum() > 0:
                self.exp_dir['left'] = True
                break
        # правая граница
        for j in range(X_len-1,X_len-bordr_min-1,-1):
            if frame[:,j].sum() > 0:
                self.exp_dir['right'] = True
                break
                
    def check_border_narr(self, frame):
        self.narr_dir = {'up':False, 'down':False, 'right':False, 'left':False}
        if frame.shape[0] <= bordr_max or frame.shape[1] <= bordr_max:
            print("Окно слишком маленькое")
            return
        self.narr_dir['up'] = not bool(frame[:bordr_max,:].sum())
        self.narr_dir['down'] = not bool(frame[-bordr_max:,:].sum())
        self.narr_dir['left'] = not bool(frame[:,:bordr_max].sum())
        self.narr_dir['right'] = not bool(frame[:,-bordr_max:].sum())
                
    def expand_borders(self):
        res = False
        if self.exp_dir['up']:
            if self.PREV[1] < self.CURR[1]:
                self.PREV = (self.PREV[0], self.PREV[1]-bordr_min)
            else:
                self.CURR = (self.CURR[0], self.CURR[1]-bordr_min)
            self._roi_img = np.vstack([np.ones((bordr_min,self._roi_img.shape[1]))*self.periph_mean, self._roi_img])
            res = True
            
        if self.exp_dir['down']:
            if self.PREV[1] > self.CURR[1]:
                self.PREV = (self.PREV[0], self.PREV[1]+bordr_min)
            else:
                self.CURR = (self.CURR[0], self.CURR[1]+bordr_min)
            self._roi_img = np.vstack([self._roi_img, np.ones((bordr_min,self._roi_img.shape[1]))*self.periph_mean])
            res = True
            
        if self.exp_dir['left']:
            if self.PREV[0] < self.CURR[0]:
                self.PREV = (self.PREV[0]-bordr_min, self.PREV[1])
            else:
                self.CURR = (self.CURR[0]-bordr_min, self.CURR[1])
            self._roi_img = np.column_stack([np.ones((self._roi_img.shape[0],bordr_min))*self.periph_mean, self._roi_img])
            res = True
            
        if self.exp_dir['right']:
            if self.PREV[0] > self.CURR[0]:
                self.PREV = (self.PREV[0]+bordr_min, self.PREV[1])
            else:
                self.CURR = (self.CURR[0]+bordr_min, self.CURR[1])
            self._roi_img = np.column_stack([self._roi_img, np.ones((self._roi_img.shape[0],bordr_min))*self.periph_mean])
            res = True
        return res
    
    def narrow_borders(self):
        res = False
        if self.narr_dir['up']:
            if self.PREV[1] < self.CURR[1]:
                self.PREV = (self.PREV[0], self.PREV[1]+bordr_min)
            else:
                self.CURR = (self.CURR[0], self.CURR[1]+bordr_min)
            self._roi_img = self._roi_img[bordr_min:,:].copy()
            res = True
            
        if self.narr_dir['down']:
            if self.PREV[1] > self.CURR[1]:
                self.PREV = (self.PREV[0], self.PREV[1]-bordr_min)
            else:
                self.CURR = (self.CURR[0], self.CURR[1]-bordr_min)
            self._roi_img = self._roi_img[:-bordr_min,:].copy()
            res = True
            
        if self.narr_dir['left']:
            if self.PREV[0] < self.CURR[0]:
                self.PREV = (self.PREV[0]+bordr_min, self.PREV[1])
            else:
                self.CURR = (self.CURR[0]+bordr_min, self.CURR[1])
            self._roi_img = self._roi_img[:,bordr_min:].copy()
            res = True
            
        if self.narr_dir['right']:
            if self.PREV[0] > self.CURR[0]:
                self.PREV = (self.PREV[0]-bordr_min, self.PREV[1])
            else:
                self.CURR = (self.CURR[0]-bordr_min, self.CURR[1])
            self._roi_img = self._roi_img[:,:-bordr_min].copy()
            res = True
        return res
        
    def roi(self):
        if not self.presentation:
            if type(self.temp_roi) != np.ndarray:
                return None
            self.temp_roi = self.get_rect_kadr(self.temp_src.copy())
            self._roi_img = self.temp_roi.copy()
        
        if type(self._roi_img) != np.ndarray:
            return None
        
        mw_scale = 1.4142
        
        t_roi_winname = "dispROI"
        cv.namedWindow(t_roi_winname, cv.WINDOW_NORMAL)
        cv.imshow(t_roi_winname, np.round(self.temp_roi).astype('uint8'))
        cv.resizeWindow(t_roi_winname, 
                        int(self.temp_roi.shape[1]/mw_scale), int(self.temp_roi.shape[0]/mw_scale))
        
        contrast_roi_img = self.contrast_img(self._roi_img)        
        cv.namedWindow(self._roi_winname, cv.WINDOW_NORMAL)
        cv.imshow(self._roi_winname, np.round(contrast_roi_img).astype('uint8'))
        cv.resizeWindow(self._roi_winname, 
                        int(contrast_roi_img.shape[1]/mw_scale), int(contrast_roi_img.shape[0]/mw_scale))
        
        frame_avg = np.average(contrast_roi_img)
        binary_roi_img = (contrast_roi_img >= (frame_avg*p_threshold)) * 255
        borders = contrast_roi_img < (frame_avg*p_threshold)
        
        self.check_border_exp(borders)
        self.check_border_narr(borders)
        self.periph_mean = np.mean(np.concatenate((self._roi_img[0,:],self._roi_img[-1,:],
                                              self._roi_img[1:-1,0],self._roi_img[1:-1,-1])))
        
        cv.namedWindow(self._binary_winname, cv.WINDOW_NORMAL)
        cv.imshow(self._binary_winname, np.round(binary_roi_img).astype('uint8'))
        cv.resizeWindow(self._binary_winname, 
                        int(binary_roi_img.shape[1]/mw_scale), int(binary_roi_img.shape[0]/mw_scale))
        return binary_roi_img
    
    def mouse_event(self, event, x, y, flags, param):
        
        if event == cv.EVENT_RBUTTONDOWN:
            self.restore_img()
            self.PREV, self.CURR = (-1,-1), (-1,-1)
            if cv.getWindowProperty(self._roi_winname, cv.WND_PROP_VISIBLE) > 0:
                cv.destroyWindow(self._roi_winname)
            if cv.getWindowProperty(self._binary_winname, cv.WND_PROP_VISIBLE) > 0:
                cv.destroyWindow(self._binary_winname)
        if self.presentation:
            return
        
        if event == cv.EVENT_LBUTTONDOWN:
            self.restore_img()
            self.PREV = (x,y)
        elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):
            self.restore_img()
            self.CURR = (x,y)
        elif event == cv.EVENT_LBUTTONUP:
            self.CURR = (x,y)
    
    def main(self):
        h, w = int(mean_kadr.shape[1]/self.scale), int(mean_kadr.shape[0]/self.scale)
        cv.namedWindow(self.WINDOW_NAME, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.WINDOW_NAME, self.mouse_event)
        while True:
            display_img = self.temp_src.copy()
            if self.CURR != (-1,-1):
                roi_res = self.roi()
                if type(roi_res) == np.ndarray:
                    minY, minX, height, width = self.get_rect_coords()
                    display_img[minY:minY+height, minX:minX+width] = roi_res
            cv.imshow(self.WINDOW_NAME, np.round(display_img).astype('uint8'))
            cv.resizeWindow(self.WINDOW_NAME, h, w) 
            key = cv.waitKey(5)
            if key == start_pres_key:
                self.presentation = True
                print(f"Start pres. Num files: {len(self.files_query)}")
            if self.presentation:
                is_exp = self.expand_borders()
                _ = self.narrow_borders()
                if len(self.files_query) == 0:
                    self.stop_presentation()
                    continue
                fname = self.files_query.pop(0)
                print(fname)
                
                new_kadr = kadrs[fname].copy()
                coef = P if not is_exp else P*P_scale
                self.temp_src = self.temp_src*(1-coef) + new_kadr*coef
                
                self.temp_roi = self.get_rect_kadr(self.temp_src.copy())
                rect_kadr = self.get_rect_kadr(new_kadr)
                
                if type(self._roi_img)!=np.ndarray:
                    print("ROI is empty!")
                    self.stop_presentation()
                    continue
                    
                new_roi_img = self._roi_img*(1-coef) + rect_kadr*coef
                check = np.vectorize(lambda x: 255. if x >= 255. else x if x > 0. else 0.)
                self._roi_img = check(new_roi_img)
                
                if cv.getWindowProperty(self._roi_winname, cv.WND_PROP_VISIBLE) > 0:
                    pass
#                 else:
#                     print("www")
#                     self.stop_presentation()
#                     continue
#                 sleep(0.2)
            
            if key == close_wins_key or cv.getWindowProperty(self.WINDOW_NAME, cv.WND_PROP_VISIBLE) < 1:
                break
            
        cv.destroyAllWindows()
        
rect = RectMouse(mean_kadr)
rect.main()