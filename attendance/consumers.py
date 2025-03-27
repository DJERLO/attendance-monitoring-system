import json
from channels.generic.websocket import AsyncWebsocketConsumer

class AttendanceConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("attendance_group", self.channel_name)
        await self.accept()
        #print("✅ WebSocket Connected!")

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("attendance_group", self.channel_name)
        #print("❌ WebSocket Disconnected")

    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data.get("message", "No message provided")

        # Send the message to all WebSocket clients
        await self.channel_layer.group_send(
            "attendance_group",
            {
                "type": "send_message",
                "message": message
            }
        )

    # Message handler (Notifies all connected clients)
    async def send_message(self, event):
        """Handles broadcasting messages to connected WebSocket clients"""
        await self.send(text_data=json.dumps({
            "message": event["message"]
        }))
    
    # Dashboard update handler
    async def send_dashboard_update(self, event):
        """Handles real-time dashboard updates"""
        data = json.loads(event["data"])
        await self.send(text_data=json.dumps(data))