from django.contrib import admin
from django.urls import path
from detector import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('detect/', views.detect_anomalies, name='detect_anomalies'),
]
