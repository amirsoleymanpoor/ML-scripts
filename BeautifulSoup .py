#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup

# URL صفحه محصولات نایک
url = "https://www.nike.com/w/mens-shoes-nik1zy7ok"

# هدر User-Agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ارسال درخواست به صفحه با هدر User-Agent
response = requests.get(url, headers=headers)

# بررسی وضعیت درخواست
if response.status_code == 200:
    # تجزیه HTML با BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # پیدا کردن محصولات
    products = soup.find_all('div', class_='product-card__body')
    
    for product in products:
        # استخراج نام محصول
        product_name = product.find('div', class_='product-card__title').text.strip()
        
        # استخراج قیمت محصول
        price_element = product.find('div', class_='product-card__price')
        if price_element:
            product_price = price_element.text.strip()
        else:
            product_price = "قیمت موجود نیست"
        
        print(f"محصول: {product_name} - قیمت: {product_price}")
else:
    print(f"خطا در بارگذاری صفحه: {response.status_code}")


# In[ ]:




