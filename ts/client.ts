const ws = new WebSocket('ws://localhost:8000/showdown/websocket');

ws.onopen = () => {
  console.log('WebSocket connection opened');
  ws.send('|/trn Mechanical Tashina|')
};

ws.onmessage = (event) => {
  const data = event.data as string;
  console.log(data);

  if (data.startsWith('|pm|')) {
    handlePrivateMessage(data);
  }
};

ws.onclose = (event) => {
  console.log('WebSocket closed:', event);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

function handlePrivateMessage(data: string) {
  const parts = data.split('|');

  console.log(parts);
  
  if (parts.length > 4 && parts[4].startsWith('/challenge')) {
    const challenger = parts[2];
    ws.send('|/utm null|')
    ws.send(`|/accept ${challenger}|`);
    console.log(`Accepted challenge from ${challenger}`);
  }
}
