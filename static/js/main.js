document.addEventListener('DOMContentLoaded', function() {
    // Code hiện tại (tooltips, card hover, v.v.) giữ nguyên...

    // Xử lý form đăng ký
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;

            const response = await fetch('/signup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, password }),
            });

            const result = await response.json();
            if (response.ok) {
                alert('Sign up successful! Please login.');
                window.location.href = '/login';
            } else {
                alert(result.message || 'Sign up failed.');
            }
        });
    }

    // Xử lý form đăng nhập
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;

            const response = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            const result = await response.json();
            if (response.ok) {
                alert('Login successful!');
                window.location.href = '/';
            } else {
                alert(result.message || 'Login failed.');
            }
        });
    }

    // Code hiện tại (quantity input, alerts, v.v.) giữ nguyên...
});