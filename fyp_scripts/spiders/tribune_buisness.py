import scrapy


class TribuneBuisnessSpider(scrapy.Spider):
    name = 'tribune_buisness'
    allowed_domains = ['tribune.com.pk']
    start_urls = ['https://tribune.com.pk/business/']

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
        'FEEDS': {'tribune_business.csv': {'format': 'csv'}}
    }


    def parse(self, response):
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse_business  , headers=self.headers )

    def parse(self, response):
        mids = response.xpath("//div[@class = 'business-featured-big-thumbnail-caption']")
        for title in mids:
            headline = title.xpath(".//a/h3/text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})

        latest_market_news = response.xpath("//div[@class = 'latest title-section']")
        for title in latest_market_news:
            headline = title.xpath(".//a/*[self::h3 or self::h4]/text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})

        featured = response.xpath("//div/div[@class = 'featured-big-thumbnail-caption']")
        for title in featured:
            headline = title.xpath(".//a/h4/text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})

        most_read = response.xpath("//div[@class = 'related-post-sdBar']/a")
        for title in most_read:
            headline = title.xpath(".//div/p/text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details_for_more, headers=self.headers, meta={'heading': headline})


        more = response.xpath("//div[@class = 'sportshortnews-caption']/a")
        for title in more:
            headline = title.xpath(".//h3/text()").get()
            headline_link = title.xpath(".//@href").get()
        
        yield response.follow(url=headline_link, callback=self.parse_details_for_more, headers=self.headers, meta={'heading': headline})

    def parse_details_for_more(self, response):
        headline = response.request.meta['heading']        
        details = response.xpath("//span[@class = 'story-text']/p[1]")

        date_time = response.xpath("//div[@class='left-authorbox']/span[2]/text()").get()
        for detail in details:
            detail_text = detail.xpath(".//text()").get()
            yield {
                'Headline' : headline,
                'Date and Time': date_time,
                'Details' : detail_text
            }


    def parse_details(self, response):
        headline = response.request.meta['heading']
        # side = response.request.meta['side']
        details = response.xpath("//p[1]/span[@style='color:black']")
        
        # details = response.xpath("//span[@class = 'story-text']/p[1]")

        date_time = response.xpath("//div[@class='left-authorbox']/span[2]/text()").get()
        for detail in details:
            detail_text = detail.xpath(".//text()").get()
            yield {
                'Headline' : headline,
                'Date and Time': date_time,
                'Details' : detail_text
            }


