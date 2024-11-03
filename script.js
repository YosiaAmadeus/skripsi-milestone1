function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const mainContent = document.getElementById("main-content");

  if (sidebar.style.left === "-250px") {
      sidebar.style.left = "0";
      mainContent.style.marginLeft = "250px";
  } else {
      sidebar.style.left = "-250px";
      mainContent.style.marginLeft = "0";
  }
}

document.addEventListener("DOMContentLoaded", function() {
  const accordions = document.querySelectorAll(".accordion");

  accordions.forEach((accordion) => {
      const header = accordion.querySelector(".accordion-header");

      header.addEventListener("click", () => {
          accordion.classList.toggle("open");
          const content = accordion.querySelector(".accordion-content");

          if (accordion.classList.contains("open")) {
              content.style.display = "block";
          } else {
              content.style.display = "none";
          }
      });
  });
});

// Fungsi untuk mengunduh file Excel
function downloadExcel() {
  window.location.href = "/download-excel";
}
