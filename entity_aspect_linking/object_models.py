from typing import List
from pykson import JsonObject, StringField, IntegerField, ListField, ObjectListField, ObjectField, Pykson, BooleanField

__author__ = "Shubham Chatterjee"
__version__ = "12/12/20"


class Location(JsonObject):
    location_id: str = StringField()
    page_id: str = StringField()
    page_title: str = StringField()
    paragraph_id: str = StringField()
    section_id: List[str] = ListField(str)
    section_headings: List[str] = ListField(str)

    def __repr__(self):
        return 'Location (\n ' \
               '    location_id: {},\n ' \
               '    page_id: {}, \n' \
               '    page_title: {},\n ' \
               '    paragraph_id: {},\n ' \
               '    section_id: {}, \n ' \
               '    section_headings: {}\n' \
               ')'.format(self.location_id, self.page_id, self.page_title, self.paragraph_id, self.section_id,
                          self.section_headings)


class Entity(JsonObject):
    entity_name: str = StringField()
    entity_id: str = StringField()
    mention: str = StringField()
    target_mention: bool = BooleanField()
    start: int = IntegerField()
    end: int = IntegerField()

    def __repr__(self):
        return 'Entity (\n' \
               '    name: {},\n' \
               '    id: {}, \n' \
               '    mention: {}, \n' \
               '    target_mention: {},\n' \
               '    start: {},\n' \
               '    end: {},\n' \
               ')'.format(self.entity_name, self.entity_id, self.mention, self.target_mention, self.start, self.end)


class AnnotatedText(JsonObject):
    content: str = StringField()
    entities: List[Entity] = ObjectListField(Entity)

    def __repr__(self):
        return 'AnnotatedText (\n' \
               '    content: {},\n' \
               '    entities: {}\n' \
               ')'.format(self.content, self.entities)


class Context(JsonObject):
    target_entity: str = StringField()
    location: Location = ObjectField(Location)
    sentence: AnnotatedText = ObjectField(AnnotatedText)
    paragraph: AnnotatedText = ObjectField(AnnotatedText)

    def __repr__(self):
        return 'Context (\n' \
               '    target_entity: {},\n' \
               '    location: {},\n' \
               '    sentence: {},\n' \
               '    paragraph: {},\n' \
               ')'.format(self.target_entity, self.location, self.sentence, self.paragraph)


class Aspect(JsonObject):
    aspect_id: str = StringField()
    aspect_name: str = StringField()
    location: Location = ObjectField(Location)
    aspect_content: AnnotatedText = ObjectField(AnnotatedText)

    def __repr__(self):
        return 'Aspect (\n' \
               '    aspect_id: {},\n' \
               '    location: {},\n' \
               '    content: {},\n' \
               '    name: {},\n' \
               ')'.format(self.aspect_id, self.location, self.aspect_content, self.aspect_name)


class AspectLinkExample(JsonObject):
    unhashed_id: str = StringField()
    id: str = StringField()
    context: Context = ObjectField(Context)
    true_aspect: str = StringField()
    candidate_aspects: List[Aspect] = ObjectListField(Aspect)

    def __repr__(self):
        return 'AspectLinkExample (\n' \
               '    unhashed_id: {},\n' \
               '    hashed_id: {},\n' \
               '    context: {},\n' \
               '    true_aspect: {},\n' \
               '    candidate_aspects: {}\n' \
               ')'.format(self.unhashed_id, self.id, self.context, self.true_aspect, self.candidate_aspects)
