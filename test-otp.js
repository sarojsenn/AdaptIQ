const http = require('http');

// Test OTP verification
const testOTPVerification = () => {
    console.log('🔢 Testing OTP verification API...');
    
    const requestData = JSON.stringify({
        email: 'test@example.com',
        otp: '123456' // This will fail but we can see the server response
    });
    
    console.log('Request data:', JSON.parse(requestData));
    
    const options = {
        hostname: 'localhost',
        port: 3000,
        path: '/api/verify-otp',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(requestData)
        }
    };
    
    const req = http.request(options, (res) => {
        console.log('📊 Status Code:', res.statusCode);
        console.log('📋 Headers:', res.headers);
        
        let body = '';
        res.on('data', (chunk) => {
            body += chunk;
        });
        
        res.on('end', () => {
            try {
                const response = JSON.parse(body);
                console.log('✅ Response:', JSON.stringify(response, null, 2));
            } catch (error) {
                console.log('📝 Raw Response:', body);
            }
        });
    });
    
    req.on('error', (error) => {
        console.log('❌ Request Error:', error.message);
    });
    
    req.write(requestData);
    req.end();
};

// Run the test
testOTPVerification();