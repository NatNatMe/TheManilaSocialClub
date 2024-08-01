document.addEventListener('DOMContentLoaded', function() {
    const allBtn = document.getElementById('all-btn');
    const tshirtBtn = document.getElementById('tshirt-btn');
    const hoodieBtn = document.getElementById('hoodie-btn');
    const searchInput = document.getElementById('search-input');
    
    const products = document.querySelectorAll('.products');

    allBtn.addEventListener('click', function() {
        products.forEach(product => {
            product.style.display = 'block';
        });
    });

    tshirtBtn.addEventListener('click', function() {
        products.forEach(product => {
            if (product.classList.contains('tshirt')) {
                product.style.display = 'block';
            } else {
                product.style.display = 'none';
            }
        });
    });

    hoodieBtn.addEventListener('click', function() {
        products.forEach(product => {
            if (product.classList.contains('hoodie')) {
                product.style.display = 'block';
            } else {
                product.style.display = 'none';
            }
        });
    });

    searchInput.addEventListener('input', function() {
        const searchTerm = searchInput.value.toLowerCase();
        products.forEach(product => {
            const productName = product.getAttribute('data-name').toLowerCase();
            if (productName.includes(searchTerm)) {
                product.style.display = 'block';
            } else {
                product.style.display = 'none';
            }
        });
    });
});
