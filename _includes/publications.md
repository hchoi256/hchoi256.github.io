## Publications

<div class="publications-list">

{% for link in site.data.publications.main %}
<div class="publication-item">

  {% if link.image %}
  <div class="publication-thumb">

    <img
      class="publication-teaser"
      src="{{ link.image | relative_url }}"
      alt="{{ link.title | escape }}"
      loading="lazy"
    >

    {% if link.badges %}
    <div class="publication-badges">
      {% for badge in link.badges %}
      <span class="publication-badge publication-badge--{{ badge.type | default: 'default' }}">
        {{ badge.text }}
      </span>
      {% endfor %}
    </div>

    {% elsif link.conference_short %}
    <div class="publication-badges">
      <span class="publication-badge publication-badge--{{ link.tag_type | default: 'default' }}">
        {{ link.conference_short }}
      </span>
    </div>
    {% endif %}

  </div>
  {% endif %}

  <div class="publication-content">

    <div class="publication-title">
      {% if link.page %}
      <a href="{{ link.page }}" target="_blank" rel="noopener noreferrer">
        <papertitle>{{ link.title }}</papertitle>
      </a>
      {% elsif link.pdf %}
      <a href="{{ link.pdf }}" target="_blank" rel="noopener noreferrer">
        <papertitle>{{ link.title }}</papertitle>
      </a>
      {% else %}
      <papertitle>{{ link.title }}</papertitle>
      {% endif %}
    </div>

    <div class="publication-authors">
      {{ link.authors }}
    </div>

    <div class="publication-venue">
      {{ link.conference }}
    </div>

    <div class="publication-links">
      {% if link.pdf %}
      <a href="{{ link.pdf }}" target="_blank" rel="noopener noreferrer">PDF</a>
      {% endif %}

      {% if link.code %}
      <a href="{{ link.code }}" target="_blank" rel="noopener noreferrer">Code</a>
      {% endif %}

      {% if link.page %}
      <a href="{{ link.page }}" target="_blank" rel="noopener noreferrer">Project Page</a>
      {% endif %}

      {% if link.bibtex %}
      <a href="{{ link.bibtex }}" target="_blank" rel="noopener noreferrer">BibTeX</a>
      {% endif %}

      {% if link.notes %}
      <span class="publication-note">{{ link.notes }}</span>
      {% endif %}

      {% if link.others %}
      {{ link.others }}
      {% endif %}
    </div>

  </div>

</div>
{% endfor %}

</div>