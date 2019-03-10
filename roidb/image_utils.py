import numpy as np
import cv2

def resize_and_pad_image(image, im_w, im_h):
    h,w=image.shape[0:2]
    scale_x=1.0*im_w/w
    scale_y=1.0*im_h/h
    '''align the longest side'''
    scale=min(scale_x, scale_y)
    
    image_scale=cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    hs,ws=image_scale.shape[0:2]
    
    image_pad=128*np.ones((im_h,im_w,3),dtype=np.uint8)
    start_x=(im_w-ws)//2
    start_y=(im_h-hs)//2
    image_pad[start_y:start_y+hs,start_x:start_x+ws,:]=image_scale[...]

    return image_pad,(start_x, start_y),scale

def compute_square_boxes(boxes, bound, gain=0.1):
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]
    
    ws=x2-x1+1
    hs=y2-y1+1
    
    cx=x1+ws*0.5
    cy=y1+hs*0.5
    
    margin=gain*(ws+hs)
    temp_sides=np.maximum(ws,hs)+margin
    temp_sides*=0.5

    nx1=np.maximum(0, cx-temp_sides)
    ny1=np.maximum(0, cy-temp_sides)
    nx2=np.minimum(bound[0], cx+temp_sides-1)
    ny2=np.minimum(bound[1], cy+temp_sides-1)
    
    return np.vstack((nx1,ny1,nx2,ny2)).transpose()

def compute_same_boxes(boxes, bound, gain=0.1):
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]
    
    ws=x2-x1+1
    hs=y2-y1+1
    
    cx=x1+ws*0.5
    cy=y1+hs*0.5
    
    margin_ws=(1+gain)*ws
    margin_hs=(1+gain)*hs
    margin_ws*=0.5
    margin_hs*=0.5

    nx1=np.maximum(0, cx-margin_ws)
    ny1=np.maximum(0, cy-margin_hs)
    nx2=np.minimum(bound[0], cx+margin_ws-1)
    ny2=np.minimum(bound[1], cy+margin_hs-1)

    return np.vstack((nx1,ny1,nx2,ny2)).transpose()

def compute_template_boxes(temp_boxes, bound, gain=0.1, shape='same'):
    if shape=='square':
        return compute_square_boxes(temp_boxes, bound, gain=gain)
    elif shape=='same':
        return compute_same_boxes(temp_boxes, bound, gain=gain)
    else:
        assert 0, 'Unrecognized shape'

'''
RPN
raw image level
'''
def calc_search_boxes(temp_boxes, bound):
    search_boxes=np.zeros((0,4),dtype=temp_boxes.dtype)
    for temp_box in temp_boxes:
        x1,y1,x2,y2=temp_box.tolist()
        cx=0.5*(x1+x2)
        cy=0.5*(y1+y2)
        w=x2-x1+1
        h=y2-y1+1
        shift_w=w*1
        shift_h=h*1
        xmin=max(0,cx-shift_w)
        ymin=max(0,cy-shift_h)
        xmax=min(bound[0], cx+shift_w-1)
        ymax=min(bound[1], cy+shift_h-1)
        search_boxes=np.append(search_boxes, np.asarray([[xmin,ymin,xmax,ymax]]), 0)
        
    return search_boxes        