import csv
from urllib.request import urlopen

import requests
from bs4 import BeautifulSoup
import re

def read_csv(csvFileName):
    name_all = []  #包含了ABC三个list
    name_A = []
    name_B = []
    name_C = []
    with open(csvFileName, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['level'] == "A":
                name_A.append(row["website"].strip())
            if row['level'] == "B":
                name_B.append(row["website"].strip())
            if row['level'] == "C":
                name_C.append(row["website"].strip())
        name_all.append(name_A)
        name_all.append(name_B)
        name_all.append(name_C)

        return name_all
def get_html_content(pageUrl:str):
    try:
        html=urlopen(pageUrl)
    except:
        return None
    else:
        return BeautifulSoup(html)

def get_html(url):

    res = requests.get(url).text
    content = BeautifulSoup(res, 'html.parser')
    return content

class article(object):

    def getInfo(url,fileName,conf,year):
        bsObj=get_html_content(url)

        fo=open(fileName+".txt","a+")
        for i in bsObj.find_all(class_="entry inproceedings"):
            title=i.find(class_="title")
            title=title.get_text()#提取文本
            if "Blockchain" in title:
                doi=i.find("a")
                doi=doi.get("href")
                doi=str(doi)
                doi=doi.replace("http://","")#除去链接头
                doi=doi.replace("https://","")
                fo.writelines(title+","+conf+","+year+","+doi+"\n")
        fo.close()





if __name__ == '__main__':
    level_list = ["A","B","C"]
    file_name_list = ["conf_A", "conf_B", "conf_C"]
    all_conf_list = read_csv("conf_csv.csv")
    print(all_conf_list)
    for index_level,level in enumerate(level_list):
        print(index_level)
        for index_conf,conf in enumerate(all_conf_list[index_level]):
            conf_html = get_html_content(conf+"index.html")
            if conf_html == None:
                pass
            print(str(index_level)+"     "+str(index_conf))
            for conf_url in conf_html.find_all(class_="toc-link"):
                eachyear_conf_url = conf_url.get("href")
                all_paper = article.getInfo(eachyear_conf_url,file_name_list[index_level],str(conf),str(conf_url))




