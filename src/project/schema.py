from graphene import ObjectType, String, Schema, Field
from project.model.crnn import CRNN_STN
from project.image import Image

crnn_model = CRNN_STN()


class QueryImage:
    @staticmethod
    def resolve(var, info, **kwargs) -> str:
        image = Image(kwargs.get('url'))

        return crnn_model.predict_text(image.get_name())


class Query(ObjectType):
    # this defines a Field `hello` in our Schema with a single Argument `name`
    text_in_image = Field(String, resolver=QueryImage.resolve,
                          url=String(required=True),
                          description='Send an image and get a text!')


schema = Schema(query=Query)
