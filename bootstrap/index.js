var fs = require('fs')
var esprima = require(__dirname + '/esprima_')

var esprima_source = fs.readFileSync(__dirname + '/esprima.js', 'utf8')
var esprima_ast = esprima.parse(esprima_source)
var esprima_serialized = JSON.stringify(esprima_ast)

fs.writeFileSync(__dirname + '/../' + 'esprima.json', esprima_serialized, 'utf8')