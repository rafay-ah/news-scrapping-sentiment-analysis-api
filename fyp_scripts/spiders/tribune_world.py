import scrapy


class TribuneWorldSpider(scrapy.Spider):
    name = 'tribune_world'
    allowed_domains = ['tribune.com.pk']
    start_urls = ['https://tribune.com.pk/world/']

    headers = {       
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": "scribe=true",
        "DNT": "1",
        "Host": "tribune.com.pk",
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
        'FEEDS': {'tribune_world.csv': {'format': 'csv'}}
    }


    def parse(self):
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse_latest  , headers=self.headers )

    def parse(self, response):
        more = response.xpath("//div[@class = 'sport-section1-right' or @class ='main-section1-caption']/a")
        for title in more:
            headline = title.xpath(".//h2/text()").get()
            headline_link = title.xpath(".//@href").get()
        
        yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})

        most_read = response.xpath("//div[@class = 'related-post-sdBar']/a")
        for title in most_read:
            headline = title.xpath(".//div/p/text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})

        mids = response.xpath("//div[@class = 'horiz-news1-caption' or @class ='sport-section1-right']/a")
        for title in mids:
            headline = title.xpath(".//h3/text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})
            



    def parse_details(self, response):
        headline = response.request.meta['heading']
        details = response.xpath("//span[@class='story-text']/p[1]")
        date_time = response.xpath("//div[@class='left-authorbox']/span[2]/text()").get()
        for detail in details:
            detail_text = detail.xpath(".//text()").get()
            yield {
                'Headline' : headline,
                'Date and Time': date_time,
                'Details' : detail_text
            }
    
    
