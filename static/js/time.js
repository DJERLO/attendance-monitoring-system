function updateTime() {
    const now = new Date();

    // Format the date and time
    const options = {
        year: 'numeric', 
        month: 'long', 
        day: '2-digit', 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true // Use 12-hour format
    };

    const timeString = now.toLocaleTimeString('en-PH', options);
    document.getElementById('currentTime').textContent = timeString;
}
// Initial call to set the time immediately, then update every 1 second
updateTime();
setInterval(updateTime, 1000);