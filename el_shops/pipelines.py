# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


from scrapy.exceptions import DropItem


class ElShopsPipeline(object):
    def process_item(self, item, spider):
        return item


class DropNonProductsPipeline(object):

    def process_item(self, item, spider):
        if item['id'] is None:
            raise DropItem("Drop empty product: %s" % item)
        return item
