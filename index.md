---
title: "baltic gallery"
layout: default
description: "Gallery of phylogenetic visualisations"
---

<section class="mt-4 mb-5">
    
    <div class="container mb-4">
        <h1 class="font-weight-bold title">Gallery</h1>    
        <div class="row">
        {% include menu-categories.html %}
        </div>
    </div>

    
<div class="container-fluid">
    
    <div class="card-columns">        
        {% for post in paginator.posts reversed %}
            {% include card-post.html %}
        {% endfor %}            
    </div>

    <div class="row justify-content-center mt-5">
     <!-- Pagination links -->
        {% if paginator.total_pages > 1 %}
        <div class="pagination"> 
          {% if paginator.previous_page %}
            <a href="{{ paginator.previous_page_path | prepend: site.baseurl | replace: '//', '/' }}">&laquo; Prev</a>
          {% else %}
            <span class="prev">&laquo; Prev</span>
          {% endif %}

          {% for page in (1..paginator.total_pages) %}
            {% if page == paginator.page %}
              <span class="webjeda">{{ page }}</span>
            {% elsif page == 1 %}
              <a href="{{site.baseurl}}/">{{ page }}</a>
            {% else %}
              <a href="{{ site.paginate_path | prepend: site.baseurl | replace: '//', '/' | replace: ':num', page }}">{{ page }}</a>
            {% endif %}
          {% endfor %}

          {% if paginator.next_page %}
            <a href="{{ paginator.next_page_path | prepend: site.baseurl | replace: '//', '/' }}">Next &raquo;</a>
          {% else %}
            <span class="next">Next &raquo;</span>
          {% endif %}
        </div>
        {% endif %}      
    </div>

</div>    
  
</section>
