from django.urls import path
from .views import ReviewListCreateAPIView

urlpatterns = [
    path('reviews/', ReviewListCreateAPIView.as_view(), name='review-list-create'),
]
