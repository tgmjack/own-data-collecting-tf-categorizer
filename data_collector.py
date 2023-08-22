from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait , Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import cv2
from tqdm import tqdm
import base64
import time
import requests


def accept_cookies(driver):
    accept_xp = '//*[@id="L2AGLb"]'
    WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,accept_xp)))
    accept = driver.find_element(By.XPATH , accept_xp)
    accept.click()
def setup_driver():
    driver_xpath2 = "chromedriver.exe"
    leaseplan_url = 'https://images.google.co.uk/'
    driver = webdriver.Chrome(driver_xpath2)
    driver.get(leaseplan_url)
#    driver.maximize_window()

    return driver

def search_for_image(driver, keyword):
    # //*[@id="APjFqb"]
    # //*[@id="APjFqb"]
    search_box_xp = '//*[@id="APjFqb"]'
    WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,search_box_xp)))
    search_box = driver.find_element(By.XPATH , search_box_xp)
    search_box.send_keys(keyword)
    time.sleep(0.5)

    search_button_xp = '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button'
    WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,search_button_xp)))
    search_button = driver.find_element(By.XPATH , search_button_xp)
    search_button.click()

    first_image_xp = '//*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img'
    WebDriverWait(driver,20).until(EC.presence_of_element_located((By.XPATH,first_image_xp)))
    first_image = driver.find_element(By.XPATH , first_image_xp)

def check_if_this_index_image_is_visible(driver, index):
    #//*[@id="islrg"]/div[1]/div[72]/a[1]/div[1]/img
    img_xp_p1 = '//*[@id="islrg"]/div[1]/div['
    img_xp_p2 = ']/a[1]/div[1]/img'
    img_xp = img_xp_p1 + str(index) + img_xp_p2
    print("looking for "+str(img_xp))
    try:
        WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,img_xp)))
        img = driver.find_element(By.XPATH , img_xp)
        return True
    except:
        return False

def scroll_down_until_this_index_is_visible(driver, index):
    times_scrolled = 0
    num_to_scroll_to = 0
    y_pixels_per_screen = 1000
    #times_scrolled = 0
   # for t in range(times_scrolled):
  #      num_to_scroll_to = times_scrolled * y_pixels_per_screen
 #       driver.execute_script("window.scrollTo(0, "+str(num_to_scroll_to)+")")
#    last_point = num_to_scroll_to
    while not check_if_this_index_image_is_visible(driver, index):
        num_to_scroll_to = times_scrolled * y_pixels_per_screen
        times_scrolled+=1
        driver.execute_script("window.scrollTo(0, "+str(num_to_scroll_to)+")")
        print("num_to_scroll_to  = "+ str(num_to_scroll_to))
        if times_scrolled%6 == 0:
            print("done")
            break;
    print("its visible")



def save_image(driver, index , category, raw_data_dir):
    print("top of save image func")
    img_xp_p1 = '//*[@id="islrg"]/div[1]/div['
    img_xp_p2 = ']/a[1]/div[1]/img'
    img_xp = img_xp_p1 + str(index) + img_xp_p2
    filename_to_save = raw_data_dir +"\\"+str(category)+"\\"+str(category)+" "+str(index)+"."
    print(filename_to_save)
    print(" ^^^^^^^^^^^^^^^^^^^^^ ")
    found_image = False
    if index < 50:
        print(img_xp)
        WebDriverWait(driver,5).until(EC.presence_of_element_located((By.XPATH,img_xp)))
        img = driver.find_element(By.XPATH , img_xp)
        found_image = True
    else:
        for i in [[51,1], [52,2], [53,3], [54,3]]: 
            if not found_image:
                img_xp = img_xp_p1+str(i[0])+"]/div[" + str(index-(50*i[1])) + img_xp_p2
                print(str(img_xp)+ "    =  img_xp to try")
                try:
                    WebDriverWait(driver,1).until(EC.presence_of_element_located((By.XPATH,img_xp)))
                    img = driver.find_element(By.XPATH , img_xp)
                    found_image = True
                except:
                    print("couldnt find image")
    if not found_image:
        raise Exception(" failed to find image ")


    data = img.get_attribute("src")
    try:
        try:
            head, encodedData = data.split(',', 1)
            file_ext = head.split(';')[0].split('/')[1]
            base = head.split(';')[1]
            decodedData = encodedData
            print(str(base)+" = base")
            if base == "base64":
                print("decoding")
                decodedData = base64.b64decode(encodedData)
                print(type(decodedData))
            elif base == "SomeOtherBase":    # Add any other base you want
                pass
            with open(filename_to_save+file_ext, 'x') as f:
            #   try:
                f.write(str(decodedData))
        except:
            response = requests.get(data)
            if response.status_code == 200:
                filename_to_save = filename_to_save+"png"
                image = Image.open(io.BytesIO(response.content))
                image.save(filename_to_save)
    except:
        print("failed to handle this one ")
    print(" :)    SAVED IMAGE !!!!!!!!!!!")


def more_button(driver):
    xp = '//*[@id="islmp"]/div/div/div[2]/div[1]/div[2]/div[2]/input'
    WebDriverWait(driver,1).until(EC.presence_of_element_located((By.XPATH,xp)))
    button = driver.find_element(By.XPATH , xp)
    button.click()

def get_data(keyword, start_index_num , end_index_num , raw_data_dir):

    driver = setup_driver()
    accept_cookies(driver)
    search_for_image(driver, keyword)



    ##### scroll a bit
    time.sleep(2)
    num_to_scroll_to = 500
    driver.execute_script("window.scrollTo(0, "+str(num_to_scroll_to)+")")
    print("   lkjjlkjlkj    ")
    time.sleep(1)


    ##### make sure its visible
    scroll_down_until_this_index_is_visible(driver, start_index_num)
    times_scrolled = 0
    
    done = False
    consequitive_fails_to_save = 0
    consequitive_fails_to_load_more = 0
    for i in range(end_index_num-start_index_num):
        if not done:
            print("up to index  =  "+str(i) )
            index = start_index_num+i
            if index % 10 == 0:  # scroll down periodically
                num_to_scroll_to = times_scrolled * 400
                times_scrolled+=1
                driver.execute_script("window.scrollTo(0, "+str(num_to_scroll_to)+")")
                time.sleep(1)
                if index%20==0: # click load more images button
                    try:
                        more_button(driver)
                        consequitive_fails_to_load_more = 0
                    except:
                        consequitive_fails_to_load_more += 1
                        print("no load more button")
                        if consequitive_fails_to_load_more > 20:   # done
                            done = True
# //*[@id="islrg"]/div[1]/div[50]/div[46]/a[1]/div[1]/img
            if index ==2:
                save_image(driver, index , keyword , raw_data_dir)
            try:
                save_image(driver, index , keyword , raw_data_dir)
                consequitive_fails_to_save = 0
            except:
                print(str(consequitive_fails_to_save)+"  =   consequitive_fails_to_save")
                consequitive_fails_to_save+=1
                if consequitive_fails_to_save > 5:
                    if consequitive_fails_to_save > 13:
                        print(9/0)
                    times_scrolled+=1
                    num_to_scroll_to = times_scrolled * 400
                    driver.execute_script("window.scrollTo(0, "+str(num_to_scroll_to)+")")
                print("failed to save image ")


    driver.quit()
