import scrapy
from scrapy.crawler import Crawler


class DawnLatestSpider(scrapy.Spider):
    name = 'dawn_latest'
    allowed_domains = ['www.dawn.com']
    start_urls = ['https://www.dawn.com/latest-news/']
    # page = 2

    headers = {       
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": "scribe=true",
        "DNT": "1",
        "Host": "www.dawn.com",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Sec-GPC": "1",
        "TE": "trailers",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; Pixel 4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.62 Mobile Safari/537.36"
    }

    custom_settings = {
       'DOWNLOAD_DELAY': 0.8,
        'FEEDS': {'dawn_latest.csv': {'format': 'csv'}}
    }

    def parse(self):
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse_latest_news  , headers=self.headers )

    def parse(self, response):
        titels = response.xpath("//h2[@data-layout = 'story']")
        for title in titels:
            headline = title.xpath(".//a/text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})

            # prev_page = response.xpath("//li[2]/a[@class='page-link']/@href").get()            
            # if prev_page:
            #     DawnLatestSpider.page -=1
            #     yield scrapy.Request(url=prev_page , callback=self.parse)
            # else: 
            #     print(" ")               
            

    def parse_details(self, response):
        headline = response.request.meta['heading']
        details = response.xpath("//div[@class = 'story__content  overflow-hidden    text-4  sm:text-4.5        pt-1  mt-1']/p[1]")
        date_time = response.xpath("//span[@class = 'timestamp--date']/text()").get()
        for detail in details:
            detail_text = detail.xpath(".//text()").get()
            yield {
                'Headline' : headline,
                'Date and Time': date_time,
                'Details' : detail_text
            }