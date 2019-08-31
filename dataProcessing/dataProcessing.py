import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from datetime import date
import configparser
import os
import zipfile
import pandas as pd
import glob
import time
from os import path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def every_downloads_chrome(driver):
    if not driver.current_url.startswith("chrome://downloads"):
        driver.get("chrome://downloads/")
    return driver.execute_script("""
        var items = downloads.Manager.get().items_;
        if (items.every(e => e.state === "COMPLETE"))
            return items.map(e => e.fileUrl || e.file_url);
        """)


def unzip(in_file_path, out_file_path):
    """

    :param in_file_path:
    :param out_file_path:
    :return:
    """
    with zipfile.ZipFile(in_file_path) as existing_zip:
        existing_zip.extractall(out_file_path)


class GetData:
    def __init__(self):
        config = configparser.ConfigParser()
        config.optionxform = str  # differ with large and small character
        # Read config
        current_dir = path.dirname(path.abspath(__file__))
        config_file = current_dir + '/data-processing-config.ini'
        abs_path_config = os.path.abspath(config_file)
        config.read(config_file, encoding="utf-8")
        # must_have_sections = ["DEFAULT"]
        # for item in must_have_sections:
        #     if item not in config.sections():
        #         print("config file not have section {}".format(item))
        #         exit()
        default = dict(config.items('DEFAULT'))
        self.url = default['url']
        self.index_id = default['index_id']
        self.start_date_id = default['start_date_id']
        self.end_date_id = default['end_date_id']
        self.driver_path = default['driver_path']
        self.submit_id = default['submit_id']
        today = date.today()
        self.today = today.strftime("%d/%m/%Y")

    def get_data_from_web(self, index, start_date, end_date, output_path):
        """
        GET DATA FOR DAILY PROCESSING MODE
        :param index:
        :param start_date:
        :param end_date:
        :param output_path:
        :return:
            downloaded file's path
        """
        # Optional argument, if not specified will search path.
        options = webdriver.ChromeOptions()
        # download option for WINDOW
        # download_option = "download.default_directory=./Data/zip"
        # options.add_argument(download_option)

        # download option for MAC
        prefs = {
            "download.default_directory": output_path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        options.add_experimental_option("prefs", prefs)

        # options.add_argument('--headless')
        options.add_argument('--disable-gpu')

        driver = webdriver.Chrome(self.driver_path, options=options)

        driver.get(self.url)

        index_element = driver.find_element_by_id(self.index_id)
        index_element.send_keys(index)

        start_date_element = driver.find_element_by_id(self.start_date_id)
        start_date_element.send_keys(start_date)

        end_date_element = driver.find_element_by_id(self.end_date_id)
        end_date_element.send_keys(end_date)

        driver.find_element_by_id(self.submit_id).click()

        # waits for all the files to be completed and returns the paths
        paths = WebDriverWait(driver, 120, 1).until(every_downloads_chrome)
        print(paths)

        # unzip file
        zip_output_path = path + "/Data/ext"
        unziped_paths = []
        for item in paths:
            item = item[7:].replace("%20", " ")
            start = start_date.replace("/", "_")
            end = end_date.replace("/", "_")
            unziped_path = zip_output_path+"/"+start+"_"+end
            # unzip file
            unzip(item, unziped_path)     # for specify time
            # unzip(item, zip_output_path)    # for daily processing
            unziped_paths.append(unziped_path)
        driver.quit()
        return unziped_paths

    @staticmethod
    def read_excel(file_paths):
        """
        READ CSV
        :param file_paths: all file_path (seperate by folder)
        :return:
            obj of Dataframe
        """
        for file_path in file_paths:
            files = glob.glob(file_path+"/*")
            columns = ["TICKER", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
            upcom = pd.DataFrame(data=None, columns=columns)
            hnx = pd.DataFrame(data=None, columns=columns)
            index = pd.DataFrame(data=None, columns=columns)
            hose = pd.DataFrame(data=None, columns=columns)
            for file in files:
                print(file)
                if file.find("UPCOM")>0:
                    temp = pd.read_excel(file, index_col=False, encoding="ISO-8859-1")
                    upcom = upcom.append(temp)
                elif file.find("HOSE")>0:
                    temp = pd.read_excel(file, index_col=False, encoding="ISO-8859-1")
                    hose = hose.append(temp)
                elif file.find("Index")>0:
                    temp = pd.read_excel(file, index_col=False, encoding="ISO-8859-1")
                    index = index.append(temp)
                elif file.find("HNX")>0:
                    temp = pd.read_excel(file, index_col=False, encoding="ISO-8859-1")
                    hnx = hnx.append(temp)
            return_obj = {
                "upcom": upcom,
                "hnx": hnx,
                "hose": hose,
                "index": index
            }
            return return_obj



if __name__ == '__main__':
    instance = GetData()
    today = date.today()
    today = today.strftime("%d/%m/%Y")
    path = os.getcwd()
    output_path = path+"/Data/zip"
    paths = instance.get_data_from_web("", "01/01/2015", today, output_path)
    print(paths)

