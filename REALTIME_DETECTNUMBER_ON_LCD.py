from paddleocr import PaddleOCR
import cv2 
import threading 
import threadpoolctl 
import concurrent 
import wandb 
import glob
import os  
import threading  
from threading import Thread 
from skimage import exposure
import numpy as np
import argparse
import imutils 
import os , sys ,  imp 
from imutils.perspective import four_point_transform   
from imutils import contours  
from concurrent.futures import ThreadPoolExecutor   
from queue import Queue
import time
import logging

ocr = PaddleOCR(lang="en") 
lock =  threading.Lock()  
queue =  Queue()  




def using_ocr_test(img_path): 
    image  =  cv2.imread(img_path)
    # ocr = PaddleOCR(lang="en")  
    try : 
        result = ocr.ocr(img_path , cls=False)   
        mat = cv2.imread(img_path)

        for line in result:
            print(line[1][0])

    except :  
        return ; 
    # for line in result:
    #     print(line[1][0])

    # # Visualization
    # mat = cv2.imread(img_path)

    # boxes = [line[0] for line in result]     

    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]

    # for box in boxes:    
    #     top_left     = (int(box[0][0]), int(box[0][1]))
    #     bottom_right = (int(box[2][0]), int(box[2][1]))

    #     cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)

    # cv2.imshow("result", mat)
    # cv2.waitKey(0)  
def using_PaddleOCR(img_path)  :   pass 

    
def using_cammera() :    
    camera= cv2.VideoCapture(0)  

    # ocr = PaddleOCR(lang="en") 
    while True : 
        _,image = camera.read() 

        # cv2.imshow("test", image)     
        img_path = "test_text.png"

        thread_1 = threading.Thread(target=using_Padd_OCR ,args = (ocr  , img_path))  
        thread_1.start() 

      
        mat = cv2.imread(img_path)  

        cv2.imshow("result", mat)
        cv2.imwrite(img_path, image)    


        thread_1.join()         

        if cv2.waitKey(1) & 0xFF == ord('s') : 
            cv2.imwrite(img_path, image)             
            break 
    camera.release()  
    cv2.destroyAllWindows()   

def using_Padd_OCR(ocr ,  img_paths ) :  

    boxes = None
    result = ocr.ocr(img_paths) 
    mat = cv2.imread(img_paths)   
    thread_2 =  threading.Thread(target=add_box(result, mat , img_paths)) 
    thread_2.start()
  
    for line in result:
        print(line[1][0])   

def add_box( result  , mat , image_path ) :  
    boxes = [line[0] for line in result]  

    for box in boxes:    
            top_left     = (int(box[0][0]), int(box[0][1]))
            bottom_right = (int(box[2][0]), int(box[2][1]))  
            cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)  

    cv2.imwrite("test_text.png", mat) 

def using_camera_2(condition_variable , queue_queue_out) : 

    # ocr = PaddleOCR(lang="en") 
    i = 0
    while (i<1000) :      
        i =  i +1   
        print( "Thread_2" ,i) 

        with condition_variable  :                      
            logging.debug('Making resource available')
            logging.debug('Notifying to all consumers')             
            condition_variable.notifyAll()

    #     _,image = camera.read()  
        
    #     img_path = "test2.png"
    #     # cv2.imshow("test", image)     

    #     thread_1 = threading.Thread(target=using_Padd_OCR ,args = (ocr  , "test_B_L.png"))  
    #     thread_1.start() 

      
    #     mat = cv2.imread("test2.png")  

    #     cv2.imshow("result", mat)
    #     cv2.imwrite("test2.png", image)    


    #     thread_1.join()         

    #     if cv2.waitKey(1) & 0xFF == ord('s') : 
    #         # cv2.imwrite(img_path, "test2.png")             
    #         break   

    # camera.release()  
    # cv2.destroyAllWindows()   

def detect_monitor( ) :                  
            
            image  = cv2.imread("288567292_3130639073842803_2542854732896822114_n.jpg")  
            cv2.imshow("test",image )
            cv2.waitKey(0)
          
        
            # args = vars(ap.parse_args())  
            # image = cv2.imread(image_path)
            ratio = image.shape[0] / 300.0
            orig = image.copy()
            image = imutils.resize(image, height = 300) 
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(gray, 30, 200)   


            cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
            screenCnt = None 

            for c in cnts : 
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.015 * peri, True)            
                if len(approx) == 4:
                        screenCnt = approx
                        break
            try : 
                cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)     
            except  :  
                print("some thing was wrong")
                # cv2.imshow("Game Boy Screen", image)                              
            cv2.imshow("Game Boy Screen", image)        
            cv2.waitKey(0)
                 
            try :
                warped = four_point_transform(image, screenCnt.reshape(4,2))   
                warped =  cv2.resize(warped, (300,200))
                cv2.imshow("wraped", warped)       
                cv2.imwrite("test_B_L.png", warped)  
                cv2.waitKey(0)

                # thread_1.join()               
                        
            except :   
                # using_camera_2()             
                warped = four_point_transform(image, screenCnt.reshape(4,2))   
                warped =  cv2.resize(warped, (300,200))
                cv2.imshow("wraped", warped)       
                cv2.imwrite("test_B_L.png", warped)
                cv2.waitKey(0)
            


            if cv2.waitKey(1) & 0xFF == ord('s') : 
                cv2.imwrite("test_B_L.png", warped)              
                return                     
                cv2.waitKey(0) 
            using_ocr_test("test_B_L.png")
def run_io_tasks_in_parallel(tasks):
    with ThreadPoolExecutor() as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            running_task.result()             


def using_paddleocr_wanb    ()  :            
        wandb.init(project="text_detecttion") ;  
        wandb.use_artifact( 'manan-goel/text_detection/model-2138qk4h:best')  
        table =  wandb.Table(columns=["Input Image", "Annotated Image"])   
        inp_imgs = sorted(glob.glob("./doc/imgs_en/*.jpg"), key=lambda x: x.split("/")[-1])
        out_imgs = sorted(glob.glob("./output/det_db/det_results/*.jpg"), key=lambda x: x.split("/")[-1]) 
                    
        wandb.init()    
        return ""                      
def consumer(queue , condition_variable) :   
 
        i = 0
        
        camera =  cv2.VideoCapture(0)                    
                
        while(True)    :
            _,image  =  camera.read()            

            i = i +  1  
            
            queue.put(i)           
        
            # args = vars(ap.parse_args())  
            # image = cv2.imread(image_path)
            ratio = image.shape[0] / 300.0
            orig = image.copy()
            image = imutils.resize(image, height = 300) 
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(gray, 30, 200)   


            cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
            screenCnt = None 

            for c in cnts : 
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.015 * peri, True)            
                if len(approx) == 4:
                        screenCnt = approx
                        break
            try : 
                cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)     
            except  :  
                print("some thing was wrong")
                # cv2.imshow("Game Boy Screen", image)                              
            cv2.imshow("Game Boy Screen", image)                         
            try :
                warped = four_point_transform(image, screenCnt.reshape(4,2))   
                warped =  cv2.resize(warped, (300,200))
                cv2.imshow("wraped", warped)       
                cv2.imwrite("test_B_L.png", warped)  
                # thread_1.join()               
                        
            except :   
                # using_camera_2()            

                print("s")  
                # warped = four_point_transform(image, screenCnt.reshape(4,2))   
                # warped =  cv2.resize(warped, (300,200))
                # cv2.imshow("wraped", warped)       
                # cv2.imwrite("test_B_L.png", warped)  
            


            if cv2.waitKey(1) & 0xFF == ord('s') : 
                cv2.imwrite("test_B_L.png", warped)              
                return                     
                cv2.waitKey(0) 
        print("Detect_monoitor in thread ")  

        with condition_variable  :   

            condition_variable.wait()  
def producer(queue ,  condition_variable) :  
    i = 0 ;    

    while(True) :   
        print("KNN in thread ")    
        data =  queue.get() 
            
        i = i +1  
        if(i != 0 ) : 
            using_ocr_test("test_B_L.png")
       
        with condition_variable  :   

            condition_variable.notifyAll()  
    


def thread_ex () : 
    # run_io_tasks_in_parallel([  
    #     lambda : detect_monitor() ,  
    #     lambda : using_camera_2() ,])  
    condition = threading.Condition()

    queue = Queue()     
    thread_1 =  threading.Thread(target=consumer , args=(queue, condition))  
    thread_2 =  threading.Thread(target=producer ,args=(queue, condition))           
    
    thread_1.start() 

    thread_2.start()      


if __name__ == '__main__':      
    # using_ocr_test("test-roi-13.png")
    #  using_camera_2()
    # using_cammera() 
    detect_monitor() 
    # thread_ex()