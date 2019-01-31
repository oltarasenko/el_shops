# -*- coding: utf-8 -*-
import predict


def handler(event, context):
    # Your code goes here!
    msg = event["message"]
    text = msg["text"]
    return predict.main(text)


if __name__ == "__main__":
    predict.main("tet")
