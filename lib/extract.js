const { readdirSync, statSync, readFileSync } = require('fs')
const { join } = require('path')

const dirs = ['cpu-log', 'cuda-log']

const obj = {}
for (const dir of dirs) {
  const files = readdirSync(dir).filter(f => f.startsWith('procs-') && f.endsWith('.out') && statSync(join(dir, f)).isFile())
  const procCollection = {}
  files.forEach(f => {
    const arr = f.split('-')
    const procs = arr[1];
    (procCollection[procs] = (procCollection[procs] || [])).push((function () {
      const contents = readFileSync(join(dir, f), 'utf8')
      let real = contents.split('\n').filter(l => l.includes('real'))[0]
      real = real.substring(real.indexOf('\t') + 1)
      return real
    })())
  })
  obj[dir] = procCollection
}

console.log(obj)