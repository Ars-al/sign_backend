from django.urls import re_path
from .consumers import GestureConsumer

websocket_urlpatterns = [
    re_path(r'ws/gesture/$', GestureConsumer.as_asgi()),
]
