import scrapy


class AryWorldSpider(scrapy.Spider):
    name = 'ary_world'
    allowed_domains = ['arynews.tv']
    start_urls = ['https://arynews.tv/category/international-2/']

    headers = {       
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": "scribe=true",
        "DNT": "1",
        "Host": "arynews.tv",
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
        'FEEDS': {'ary_world.csv': {'format': 'csv'}}
    }


    def parse(self, response):
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse_latest  , headers=self.headers )

    def parse(self, response):
        titles = response.xpath("//h3[@class='entry-title td-module-title']/a")
        for title in titles:
            headline = title.xpath(".//text()").get()
            headline_link = title.xpath(".//@href").get()

            yield response.follow(url=headline_link, callback=self.parse_details, headers=self.headers, meta={'heading': headline})

    def parse_details(self, response):
        headline = response.request.meta['heading']
        details = response.xpath("//div[@class = 'tdb-block-inner td-fix-index']/*[self::p[1] or self::p[@data-testid='paragraph-0']]/strong")
        date_time = response.xpath("//div[@class='td_block_wrap tdb_single_date tdi_67 td-pb-border-top td_block_template_1 tdb-post-meta']/div/time/text()").get()
        for detail in details:
            detail_text = detail.xpath(".//text()").get()
            yield {
                'Headline' : headline,
                'Date and Time': date_time,
                'Details' : detail_text
            }
