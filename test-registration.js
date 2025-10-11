const http = require('http');

const postData = JSON.stringify({
    firstName: 'Test',
    lastName: 'User',
    email: 'test@example.com',
    password: 'testpass123'
});

const options = {
    hostname: 'localhost',
    port: 3000,
    path: '/api/register',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
    }
};

console.log('🧪 Testing registration API...');
console.log('Request data:', JSON.parse(postData));

const req = http.request(options, (res) => {
    console.log(`📊 Status Code: ${res.statusCode}`);
    console.log(`📋 Headers:`, res.headers);

    let data = '';
    res.on('data', (chunk) => {
        data += chunk;
    });

    res.on('end', () => {
        try {
            const response = JSON.parse(data);
            console.log('✅ Response:', JSON.stringify(response, null, 2));
        } catch (e) {
            console.log('📄 Raw Response:', data);
        }
    });
});

req.on('error', (e) => {
    console.error('❌ Request Error:', e.message);
});

req.write(postData);
req.end();