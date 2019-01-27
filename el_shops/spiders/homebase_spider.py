import scrapy
from el_shops.items import Product


class HomeBaseSpider(scrapy.Spider):
    name = "homebase"

    def start_requests(self):
        urls = [
            "https://www.homebase.co.uk/our-range/lighting-and-electrical",
            "https://www.homebase.co.uk/yale-wired-outdoor-dome-camera-hd1080_p484127"
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        id = response.css(".product-in::attr(content)").extract_first()
        name = response.css(".page-title h1::text").extract_first()
        price = response.css(".price-value::attr(content)").extract_first()
        descr = response.css(
            "#tab-description .product-details__description::text").extract_first()
        url = response.url
        image_urls = response.css("img.rsTmb::attr(src)").extract()

        yield Product(id=id, name=name, price=price, descr=descr, url=url, image_urls=image_urls)

        urls = response.css("a.product-list__link::attr(href)").extract()
        for url in urls:
            next_page = response.urljoin(url)
            yield scrapy.Request(url=next_page, callback=self.parse)
