import scrapy


# class DawnuWorldSpider(scrapy.Spider):
#     name = 'dawnu_world'
#     allowed_domains = ['www.dawnnews.tv/world']
#     start_urls = ['http://www.dawnnews.tv/world/']

#     def parse(self, response):
#         pass

class DawnUrduLatestSpider(scrapy.Spider):
    name = 'dawnu_world'
    # allowed_domains = ['www.dawnnews.tv']
    # start_urls = ['http://www.dawnnews.tv/']

    allowed_domains = ['www.dawnnews.tv']
    # start_urls = ['https://www.dawn.com/archive/2022-02-09']
    # url = ['https://www.dawn.com']
    # page = 1
    
    
    headers = {       
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": "scribe=true",
        "DNT": "1",
        "Host": "www.dawnnews.tv",
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
        'FEEDS': {'dawnUrdu_business.csv': {'format': 'csv'}}
    }


    def start_requests(self):
        yield scrapy.Request(url='https://www.dawnnews.tv/world', callback=self.parse, headers=self.headers)

    def parse(self, response):
        # titles = response.xpath("//h2[@class = 'story__title      text-6.5  text-colorsDawn-green hover:text-colorsDawn-red-darkest font-bold leading-tight    pt-0.75  pb-2  ']/a")
        titles = response.xpath("//h2[@data-layout='story']/a")

        for title in titles:
            headline = title.xpath(".//text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link,  callback=self.parse_headline, meta={'heading': headline}, headers=self.headers)

        # side_market_news = response.xpath("//div[@class = 'text-4  px-2  mb-2']/a")
        # for title in side_market_news:
        #     headline = title.xpath(".//text()").get()
        #     headline_link = title.xpath(".//@href").get()

        #     yield response.follow(url=headline_link,  callback=self.parse_headline, meta={'heading': headline}, headers=self.headers)

    def parse_headline(self, response):
        headline = response.request.meta['heading']
        # logging.info(response.url)
        full_detail = response.xpath("//div[@class = 'story__content  overflow-hidden    text-5.5        pt-0  sm:pt-6.5']/p[1]")
        date_and_time = response.xpath("//span[@class = 'story__time    text-4    ']/text()").get()

        for detail in full_detail:
            data = detail.xpath(".//text()").get()
            yield {
                'headline': headline,
                'date_and_time': date_and_time,
                'details': data
            }