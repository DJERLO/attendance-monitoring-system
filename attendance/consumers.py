import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import Employee

class AttendanceConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("attendance_group", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("attendance_group", self.channel_name)

    async def attendance_update(self, event):
        await self.send(text_data=json.dumps(event["data"]))

# Notification Consumer
# This consumer handles real-time notifications for the users.
# It listens for new notifications and sends them to connected clients.
class NotificationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Extract employee ID from the URL route or querystring (e.g., self.scope["user"].id)
        user = self.scope["user"]
        employee = await Employee.objects.select_related('user').aget(user=user)

        if not employee:
            # If the user is not an employee, reject the connection
            await self.close()
            return
        
        # Unique group for individual notifications
        self.group_name = f'notifications_{employee.id}'

        # Global group for all employees (for announcements)
        self.global_group = "notifications_all"
        
        # Add the user to both the individual and global groups
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.channel_layer.group_add(self.global_group, self.channel_name)
        
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)
        await self.channel_layer.group_discard(self.global_group, self.channel_name)

    async def receive(self, text_data):
        pass  # We only send from backend

    async def send_notification(self, event):
        await self.send(text_data=json.dumps(event["data"]))