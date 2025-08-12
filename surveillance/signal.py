from django.db.models.signals import post_save
from django.dispatch import receiver
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.apps import apps  # ✅ import apps

@receiver(post_save)
def notify_log_created(sender, instance, created, **kwargs):
    Logs = apps.get_model('surveillance', 'Logs')  # ✅ safely get model after registry loads

    if sender != Logs:
        return  # ❌ ignore signals from other models

    if created:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            "logs",
            {
                "type": "send_log_notification",
                "data": {
                    "text": f"{instance.person} or {instance.weapon} was seen in {instance.camera}",
                    "timestamp": str(instance.date)
                }
            }
        )
