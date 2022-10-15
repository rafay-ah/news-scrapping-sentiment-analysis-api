from turtle import title
import scrapy


class TribuneurduWorldSpider(scrapy.Spider):
    name = 'tribuneUrdu_world'
    allowed_domains = ['www.express.pk']
    start_urls = ['https://www.express.pk/world/']


    headers = {       
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": "scribe=true",
        "DNT": "1",
        "Host": "www.express.pk",
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
        'FEEDS': {'tribuneUrdu_world.csv': {'format': 'csv'}}
    }


    def parse(self, response):
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse_world  , headers=self.headers )
        
    def parse_world(self, response):
        titles = response.xpath("//h1[@class='title ']/a")
        for title in titles:
            headline = title.xpath(".//text()").get()
            headline_link = title.xpath(".//@href").get()
            
            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})
            
        titles_N = response.xpath("//div[@class='cstoreyitem']/div/a[2]")
        for title in titles_N:
            headline = title.xpath(".//text()").get()
            headline_link = title.xpath(".//@href").get()
            
            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})
            
        # titles_F = response.xpath("//div[@class = 'cstoreyitem  ']/div")
        # for title in titles_F:
        #     headline = title.xpath(".//p/text()").get()
        #     headline_link = title.xpath(".//a[@class = 'title']/@href").get()
            
        #     yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})
            
    def parse_details(self, response):
        headline = response.request.meta['heading']
        # details = response.xpath("//p[@ style='text-align: right;']/strong")
        details = response.xpath("//p/strong/text()").getall()
        date_time = response.xpath("//span[@class = 'timestamp']/text()").get()
        for detail in details:
            # detail_text = detail.xpath(".//text()").get()
            yield {
                'Headline' : headline,
                'Date and Time': date_time,
                # 'Details' : detail_text
                'Details' : details
            }
        