/**
 * Global App Script for DeepShield AI
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Global Ripple Effect
  if (window.initRippleEffect) {
    window.initRippleEffect();
  }

  // Initialize Scroll Observer
  if (window.initScrollObserver) {
    window.initScrollObserver();
  }

  // Mobile Navigation Drawer Toggle
  const hamburger = document.querySelector('.hamburger');
  const mobileNav = document.querySelector('.mobile-nav');

  if (hamburger && mobileNav) {
    hamburger.addEventListener('click', () => {
      mobileNav.classList.toggle('open');
      const isOpen = mobileNav.classList.contains('open');
      hamburger.setAttribute('aria-expanded', isOpen);
    });

    // Close mobile nav on link click
    mobileNav.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        mobileNav.classList.remove('open');
      });
    });
  }

  // Active Link Highlighter
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll('.nav-link, .mobile-nav a');

  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href) {
      // Normalize paths
      if (currentPath.endsWith(href) || (currentPath === '/' && href.includes('index.html'))) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    }
  });

  // Modal Backdrop Click Listener
  document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        overlay.classList.remove('open');
      }
    });
  });

  // Modal Close Buttons
  document.querySelectorAll('[data-close-modal]').forEach(btn => {
    btn.addEventListener('click', () => {
      const modalId = btn.getAttribute('data-close-modal');
      if (window.closeModal) {
        window.closeModal(modalId);
      }
    });
  });
});
