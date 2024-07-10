from django.shortcuts import render
from rest_framework import generics
from .models import Category, MenuItem, Cart, Order
from .serializers import CategorySerializer, MenuItemSerializer, CartSerializer, OrderSerializer
from django.contrib.auth.models import User, Group
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser

class CategoryList(generics.ListCreateAPIView):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer

class MenuItemList(generics.ListCreateAPIView):
    queryset = MenuItem.objects.all()
    serializer_class = MenuItemSerializer

class CartList(generics.ListCreateAPIView):
    queryset = Cart.objects.all()
    serializer_class = CartSerializer

class OrderList(generics.ListCreateAPIView):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer

@api_view(['POST'])
def add_user_to_group(request):
    user_id = request.data.get('user_id')
    group_name = request.data.get('group_name')
    try:
        user = User.objects.get(id=user_id)
        group, created = Group.objects.get_or_create(name=group_name)
        group.user_set.add(user)
        return Response({'message': 'User added to group successfully.'}, status=status.HTTP_200_OK)
    except User.DoesNotExist:
        return Response({'error': 'User does not exist.'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['POST'])
def remove_user_from_group(request):
    user_id = request.data.get('user_id')
    group_name = request.data.get('group_name')
    try:
        user = User.objects.get(id=user_id)
        group = Group.objects.get(name=group_name)
        group.user_set.remove(user)
        return Response({'message': 'User removed from group successfully.'}, status=status.HTTP_200_OK)
    except User.DoesNotExist:
        return Response({'error': 'User does not exist.'}, status=status.HTTP_404_NOT_FOUND)
    except Group.DoesNotExist:
        return Response({'error': 'Group does not exist.'}, status=status.HTTP_404_NOT_FOUND)