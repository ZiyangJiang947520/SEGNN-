from django.urls import path
from . import views
from .views import add_user_to_group, remove_user_from_group
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
urlpatterns = [
    path('categories/', views.CategoryList.as_view(), name='category-list'),
    path('menu-items/', views.MenuItemList.as_view(), name='menuitem-list'),
    path('carts/', views.CartList.as_view(), name='cart-list'),
    path('orders/', views.OrderList.as_view(), name='order-list'),
    path('add-user-to-group/', add_user_to_group, name='add-user-to-group'),
    path('remove-user-from-group/', remove_user_from_group, name='remove-user-from-group'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    # 添加其他需要的路径
]
