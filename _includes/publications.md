<hr class="section-divider">

## Publications

<style>
.publication-filter input[type="radio"] {
  display: none;
}

.publication-filter-tabs {
  margin-top: 4px;
  margin-bottom: 18px;
  font-size: 15px;
  font-weight: 600;
}

.publication-filter-tabs label {
  cursor: pointer;
  color: #777;
  padding: 0 2px;
}

.publication-filter-separator {
  color: #bbb;
  margin: 0 6px;
}

.publication-panel {
  display: none;
}

#pub-filter-selected:checked ~ .publication-filter-tabs label[for="pub-filter-selected"],
#pub-filter-all:checked ~ .publication-filter-tabs label[for="pub-filter-all"] {
  color: #111;
  font-weight: 700;
  text-decoration: underline;
  text-underline-offset: 4px;
}

#pub-filter-selected:checked ~ .publication-panel-selected {
  display: block;
}

#pub-filter-all:checked ~ .publication-panel-all {
  display: block;
}
</style>

<div class="publication-filter">

  <input type="radio" id="pub-filter-selected" name="pub-filter" checked>
  <input type="radio" id="pub-filter-all" name="pub-filter">

  <div class="publication-filter-tabs">
    <label for="pub-filter-selected">Selected</label>
    <span class="publication-filter-separator">/</span>
    <label for="pub-filter-all">All</label>
  </div>

  <div class="publication-panel publication-panel-selected">
    <div class="publications-list">

{% for link in site.data.publications.main %}
{% if link.selected %}
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
      <papertitle>{{ link.title }}</papertitle>
    </div>

    <div class="publication-authors">
      {{ link.authors }}
    </div>

    {% if link.special_note %}
    <div class="publication-special-note">
      {{ link.special_note }}
    </div>
    {% endif %}

    <div class="publication-venue">
      {{ link.conference }}
    </div>

    <div class="publication-links">
      {% if link.demo %}
      <a href="{{ link.demo }}" target="_blank" rel="noopener noreferrer">Demo</a>
      {% endif %}

      {% if link.pdf %}
      <a href="{{ link.pdf }}" target="_blank" rel="noopener noreferrer">Paper</a>
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
{% endif %}
{% endfor %}

    </div>
  </div>

  <div class="publication-panel publication-panel-all">
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
      <papertitle>{{ link.title }}</papertitle>
    </div>

    <div class="publication-authors">
      {{ link.authors }}
    </div>

    {% if link.special_note %}
    <div class="publication-special-note">
      {{ link.special_note }}
    </div>
    {% endif %}

    <div class="publication-venue">
      {{ link.conference }}
    </div>

    <div class="publication-links">
      {% if link.demo %}
      <a href="{{ link.demo }}" target="_blank" rel="noopener noreferrer">Demo</a>
      {% endif %}

      {% if link.pdf %}
      <a href="{{ link.pdf }}" target="_blank" rel="noopener noreferrer">Paper</a>
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
  </div>

</div>
