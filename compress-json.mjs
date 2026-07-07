import { readFile, writeFile } from 'node:fs/promises';

const suffix = '_Trimmed';

async function main() {
  const avatarData = JSON.parse(await readFile('AvatarExcelConfigData.json', 'utf-8'));
  const fettersData = JSON.parse(await readFile('FettersExcelConfigData.json', 'utf-8'));
  const textMap = JSON.parse(await readFile('TextMap_MediumCHS.json', 'utf-8'));

  const usedHashes = new Set();
  for (const a of avatarData) {
    if (a.nameTextMapHash) usedHashes.add(String(a.nameTextMapHash));
  }
  for (const f of fettersData) {
    if (f.voiceTitleTextMapHash) usedHashes.add(String(f.voiceTitleTextMapHash));
  }

  console.log(`引用到的 textMap 条目数: ${usedHashes.size}`);

  const trimmedAvatar = avatarData
    .filter(a => a.nameTextMapHash)
    .map(a => ({ id: a.id, nameTextMapHash: a.nameTextMapHash }));
  console.log(`Avatar 条目: ${avatarData.length} -> ${trimmedAvatar.length}`);

  const trimmedFetters = fettersData
    .filter(f => f.voiceTitleTextMapHash && f.avatarId)
    .map(f => ({ avatarId: f.avatarId, voiceTitleTextMapHash: f.voiceTitleTextMapHash }));
  console.log(`Fetters 条目: ${fettersData.length} -> ${trimmedFetters.length}`);

  const trimmedTextMap = {};
  for (const hash of usedHashes) {
    if (textMap[hash] !== undefined) {
      trimmedTextMap[hash] = textMap[hash];
    }
  }
  console.log(`TextMap 条目: ${Object.keys(textMap).length} -> ${Object.keys(trimmedTextMap).length}`);

  await Promise.all([
    writeFile(`AvatarExcelConfigData${suffix}.json`, JSON.stringify(trimmedAvatar)),
    writeFile(`FettersExcelConfigData${suffix}.json`, JSON.stringify(trimmedFetters)),
    writeFile(`TextMap_MediumCHS${suffix}.json`, JSON.stringify(trimmedTextMap)),
  ]);

  console.log('✅ 全部完成');
}

main().catch(console.error);
