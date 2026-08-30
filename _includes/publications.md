<hr class="section-divider">

## Publications

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
  </div>

</div>
