const http = require('http');

// Test registration with a unique email
const testRegistration = () => {
    console.log('🧪 Testing registration with unique email...');
    
    const timestamp = Date.now();
    const requestData = JSON.stringify({
        firstName: 'John',
        lastName: 'Doe', 
        email: `test${timestamp}@example.com`,
        password: 'securepass123'
    });
    
    console.log('Request data:', JSON.parse(requestData));
    
    const options = {
        hostname: 'localhost',
        port: 3000,
        path: '/api/register',
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
                
                if (response.success && response.userId) {
                    console.log('\n🔍 Now testing OTP verification with this user...');
                    testOTPWithUser(response.userId, JSON.parse(requestData).email);
                }
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

// Test OTP verification with the created user
const testOTPWithUser = (userId, email) => {
    setTimeout(() => {
        console.log('\n🔢 Testing OTP verification for user:', userId);
        
        const requestData = JSON.stringify({
            userId: userId,
            otp: '123456' // Wrong OTP to test the validation
        });
        
        console.log('OTP Request data:', JSON.parse(requestData));
        
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
            console.log('📊 OTP Status Code:', res.statusCode);
            
            let body = '';
            res.on('data', (chunk) => {
                body += chunk;
            });
            
            res.on('end', () => {
                try {
                    const response = JSON.parse(body);
                    console.log('✅ OTP Response:', JSON.stringify(response, null, 2));
                } catch (error) {
                    console.log('📝 Raw OTP Response:', body);
                }
            });
        });
        
        req.on('error', (error) => {
            console.log('❌ OTP Request Error:', error.message);
        });
        
        req.write(requestData);
        req.end();
    }, 1000);
};

// Run the test
testRegistration();