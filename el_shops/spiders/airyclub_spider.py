import scrapy
from el_shops.items import Product


class AiryclubSpider(scrapy.Spider):
    name = "airyclub"

    def start_requests(self):
        urls = [
            "https://www.airyclub.com/Electronics-r9874/",
            "https://www.airyclub.com/Shoes-r9880/",
            "https://www.airyclub.com/Smart-Band-Watch-Bracelet-Wristband-Fitness-Tracker-Blood-Pressure-GSN1547535164710993218425982-g4755465-m5778381"
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        id = response.css(".icon-list span::attr(data-id)").extract_first()
        name = response.css('h1.prod-name::text').extract_first()
        price = response.css('#prod-shop-price::text').extract_first()
        descr = response.css('xmp').extract_first()
        url = response.url
        images = response.css(".thumblist img::attr(src)").extract()
        image_urls = ["https:%s" % img for img in images]

        yield Product(id=id, name=name, price=price, descr=descr, url=url, image_urls=image_urls)

        urls = response.css(".grid-link-image a::attr(href)").extract()
        for url in urls:
            next_page = response.urljoin(url)
            yield scrapy.Request(url=next_page, callback=self.parse)
