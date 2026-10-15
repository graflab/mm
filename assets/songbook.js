const search = document.querySelector('#song-search');
const clear = document.querySelector('#clear-search');
const songs = [...document.querySelectorAll('.song')];
const count = document.querySelector('#song-count');
const empty = document.querySelector('#no-results');

function filterSongs() {
  const query = search.value.trim().toLocaleLowerCase();
  let visible = 0;
  for (const song of songs) {
    song.hidden = !song.querySelector('.song-title').textContent.toLocaleLowerCase().includes(query);
    if (!song.hidden) visible++;
  }
  count.textContent = query ? `${visible} of ${songs.length} songs` : `${songs.length} songs · A–Z`;
  empty.hidden = visible !== 0;
  clear.hidden = search.value.length === 0;
}

function openLinkedSong() {
  const song = songs.find(item => `#${item.id}` === window.location.hash);
  if (!song) return;
  search.value = '';
  filterSongs();
  song.open = true;
  requestAnimationFrame(() => song.scrollIntoView({ block: 'start' }));
}

search.addEventListener('input', filterSongs);
clear.addEventListener('click', () => {
  search.value = '';
  filterSongs();
  search.focus();
});
window.addEventListener('hashchange', openLinkedSong);
document.querySelector('.search-tools').hidden = false;
filterSongs();
openLinkedSong();
