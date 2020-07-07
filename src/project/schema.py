from graphene import ObjectType, String, Schema, Field


class QueryImage:
    @staticmethod
    def resolve(var, info, **kwargs) -> str:
        return kwargs.get('url')


class Query(ObjectType):
    # this defines a Field `hello` in our Schema with a single Argument `name`
    text_in_image = Field(String, resolver=QueryImage.resolve,
                          url=String(required=True),
                          description='Send an image and get a text!')


schema = Schema(query=Query)
