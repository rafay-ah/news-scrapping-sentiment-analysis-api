import scrapy


# class AryuBusinessSpider(scrapy.Spider):
#     name = 'aryu_business'
#     allowed_domains = ['urdu.arynews.tv/category/business']
#     start_urls = ['http://urdu.arynews.tv/category/business/']

#     def parse(self, response):
#         pass


class AryLatestSpider(scrapy.Spider):
    name = 'aryu_business'
    allowed_domains = ['urdu.arynews.tv']
    start_urls = ['https://urdu.arynews.tv/category/business']

    headers = {       
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": "scribe=true",
        "DNT": "1",
        "Host": "urdu.arynews.tv",
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
        'FEEDS': {'aryUrdu_business.csv': {'format': 'csv'}}
    }


    def parse(self, response):
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse_latest  , headers=self.headers )

    def parse(self, response):
        titles = response.xpath("//h2[@class = 'title']/a")
        for title in titles:
            headline = title.xpath(".//text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})
            
        # prev_page = response.xpath("//a[@class='btn-bs-pagination' or @class='next page-numbers']/@href").get()
        # # prev = 'https://www.dawn.com' + str(prev_page)
        # for i in range(0,1):
        #     yield scrapy.Request(url=prev_page, callback=self.parse, headers=self.headers)

    def parse_details(self, response):
        headline = response.request.meta['heading']
        # details = response.xpath("//p[@ style='text-align: right;']/strong")
        details = response.xpath("//*[self::p/strong or self::p[@ style='text-align: right;']/strong or self::p[@style='direction: rtl']/strong or self::span[@style='color: #000000;']/strong]")
        date_time = response.xpath("//time/b/text()").get()
        for detail in details:
            detail_text = detail.xpath(".//text()").get()
            yield {
                'Headline' : headline,
                'Date and Time': date_time,
                'Details' : detail_text
            }