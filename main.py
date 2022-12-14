from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

# 以bs对象返回输入的url
def getBsObj(pageUrl:str):
    try:
        html=urlopen(pageUrl)
    except:
        return None
    else:
        return BeautifulSoup(html)


if __name__ == "__main__":

    # 输入期刊或会议的网址
    bsObj = getBsObj("https://dblp.uni-trier.de/db/journals/comcom/index.html")

    # 会议的每个volume的链接的相似点是，文本都以volume开头，用正则表达式进行筛选
    list = bsObj.find_all('a',text=re.compile("Volume.*"))
    for i in list:
        herf = i.get('href')  #返回这个页面中符合条件的所有链接
        print(herf)
