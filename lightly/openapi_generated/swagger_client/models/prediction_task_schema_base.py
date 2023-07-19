# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


from __future__ import annotations
import pprint
import re  # noqa: F401
import json
import lightly.openapi_generated.swagger_client.models



from pydantic import Extra,  BaseModel, Field, StrictStr, constr, validator

class PredictionTaskSchemaBase(BaseModel):
    """
    The schema for predictions or labels when doing classification, object detection, keypoint detection or instance segmentation 
    """
    name: constr(strict=True, min_length=1) = Field(..., description="A name which is safe to have as a file/folder name in a file system")
    type: StrictStr = Field(..., description="This is the TaskType. Due to openapi.oneOf fuckery with discriminators, this needs to be a string")
    __properties = ["name", "type"]

    @validator('name')
    def name_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 ._-]+$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 ._-]+$/")
        return value

    class Config:
        """Pydantic configuration"""
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = Extra.forbid

    # JSON field name that stores the object type
    __discriminator_property_name = 'type'

    # discriminator mappings
    __discriminator_value_class_map = {
        'PredictionTaskSchemaKeypoint': 'PredictionTaskSchemaKeypoint',
        'PredictionTaskSchemaSimple': 'PredictionTaskSchemaSimple'
    }

    @classmethod
    def get_discriminator_value(cls, obj: dict) -> str:
        """Returns the discriminator value (object type) of the data"""
        discriminator_value = obj[cls.__discriminator_property_name]
        if discriminator_value:
            return cls.__discriminator_value_class_map.get(discriminator_value)
        else:
            return None

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.dict(by_alias=by_alias))

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the model"""
        return json.dumps(self.to_dict(by_alias=by_alias))

    @classmethod
    def from_json(cls, json_str: str) -> Union(PredictionTaskSchemaKeypoint, PredictionTaskSchemaSimple):
        """Create an instance of PredictionTaskSchemaBase from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> Union(PredictionTaskSchemaKeypoint, PredictionTaskSchemaSimple):
        """Create an instance of PredictionTaskSchemaBase from a dict"""
        # look up the object type based on discriminator mapping
        object_type = cls.get_discriminator_value(obj)
        if object_type:
            klass = getattr(lightly.openapi_generated.swagger_client.models, object_type)
            return klass.from_dict(obj)
        else:
            raise ValueError("PredictionTaskSchemaBase failed to lookup discriminator value from " +
                             json.dumps(obj) + ". Discriminator property name: " + cls.__discriminator_property_name +
                             ", mapping: " + json.dumps(cls.__discriminator_value_class_map))

