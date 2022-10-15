import scrapy


class DawnBuisnessSpider(scrapy.Spider):
    name = 'dawn_business'
    allowed_domains = ['www.dawn.com']
    start_urls = ['https://www.dawn.com/business/']

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
        'FEEDS': {'dawn_business.csv': {'format': 'csv'}}
    }

    def parse(self, response):
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse_business  , headers=self.headers )

    def parse(self, response):
        titels = response.xpath("//h2[@data-layout = 'story']")
        for title in titels:
            headline = title.xpath(".//a/text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})

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
