#!/usr/bin/env node

/**
 * Generate app icons from SVG logo
 * Creates icon.png (1024x1024) for electron-builder
 */

const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const logoSvg = fs.readFileSync(path.join(__dirname, 'resources/logo.svg'));

async function generateIcons() {
  console.log('🎨 Generating app icons from logo.svg...');
  
  try {
    // Generate 1024x1024 icon.png for electron-builder
    await sharp(logoSvg)
      .resize(1024, 1024)
      .png()
      .toFile(path.join(__dirname, 'resources/icon.png'));
    
    console.log('✅ Created resources/icon.png (1024x1024)');
    
    // Generate favicon sizes
    await sharp(logoSvg)
      .resize(32, 32)
      .png()
      .toFile(path.join(__dirname, 'public/favicon-32.png'));
    
    console.log('✅ Created public/favicon-32.png');
    
    await sharp(logoSvg)
      .resize(16, 16)
      .png()
      .toFile(path.join(__dirname, 'public/favicon-16.png'));
    
    console.log('✅ Created public/favicon-16.png');
    
    console.log('\n🎉 All icons generated successfully!');
  } catch (error) {
    console.error('❌ Error generating icons:', error);
    process.exit(1);
  }
}

generateIcons();
