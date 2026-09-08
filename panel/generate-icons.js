#!/usr/bin/env node

/**
 * Derive renderer icons from the supplied macOS app icon.
 * resources/icon.png is the canonical 1024px source used by electron-builder.
 */

const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const appIcon = fs.readFileSync(path.join(__dirname, 'resources/icon.png'));

async function generateIcons() {
  console.log('Generating renderer icons from resources/icon.png...');

  try {
    const metadata = await sharp(appIcon).metadata();
    if (metadata.width !== 1024 || metadata.height !== 1024) {
      throw new Error('resources/icon.png must be the 1024x1024 macOS source icon');
    }
    const output = path.join(__dirname, 'src/renderer/public');
    fs.mkdirSync(output, { recursive: true });
    for (const [size, name] of [[16, 'favicon-16.png'], [32, 'favicon-32.png'], [64, 'app-icon-64.png']]) {
      await sharp(appIcon).resize(size, size).png().toFile(path.join(output, name));
      console.log(`Created ${name}`);
    }

    console.log('\n🎉 All icons generated successfully!');
  } catch (error) {
    console.error('❌ Error generating icons:', error);
    process.exit(1);
  }
}

generateIcons();
