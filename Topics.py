#!/Python27/python
# -*- coding: UTF-8 -*-


class Topics:

    def __init__(self, topics):
        self.topics = topics

    def clean_topics(self):
        for index, topic in enumerate(self.topics):
            topic = topic.encode('utf8')
            for i in range(0, 10):
                topic = topic.replace(str(i), "")
            topic = topic.replace(" ", "")
            topic = topic.replace(".", "")
            topic = topic.replace("*", "")
            topic = topic.replace("+", " ")
            topic = topic.split(" ")
            self.topics[index] = topic
        return self.topics

    def get_string_for(self, i):
        if i in self.topics:
            topic = " ".join(self.topics[i])
            return topic
        else:
            return ""
