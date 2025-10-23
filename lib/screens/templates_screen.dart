import 'dart:math';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/api_service.dart';

class TemplatesScreen extends StatefulWidget {
  const TemplatesScreen({super.key});
  @override
  State<TemplatesScreen> createState() => _TemplatesScreenState();
}

class _TemplatesScreenState extends State<TemplatesScreen> {
  List<dynamic> _templates = [];
  bool _loading = true;
  String _query = '';

  @override
  void initState() {
    super.initState();
    _fetch();
  }

  Future<void> _fetch() async {
    setState(() => _loading = true);
    try {
      final res = await ApiService.get('/api/templates');
      setState(() { _templates = res is List ? res : []; });
    } catch (e) {
      // fallback: generate demo
      final rnd = Random();
      _templates = List.generate(8, (i)=> {
        'id':'t$i','title':'Demo Template #$i','category':i%2==0?'Motivation':'Business','thumbnail':'https://picsum.photos/seed/t$i/400/600'
      });
    } finally { setState(()=>_loading=false); }
  }

  void _useTemplate(Map t) {
    // send to home - in this simple flow we navigate back with result
    Navigator.pop(context, t);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Templates'), backgroundColor: Colors.deepPurple),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: TextField(
              onChanged: (v)=> setState(()=>_query=v),
              decoration: InputDecoration(
                hintText: 'Search templates...',
                prefixIcon: const Icon(Icons.search),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
          Expanded(
            child: _loading ? const Center(child: CircularProgressIndicator()) :
            GridView.builder(
              padding: const EdgeInsets.all(12),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 2, childAspectRatio: 9/16, crossAxisSpacing: 12, mainAxisSpacing: 12),
              itemCount: _templates.length,
              itemBuilder: (_,i){
                final t = _templates[i];
                if (_query.isNotEmpty && !(t['title']?.toString().toLowerCase() ?? '').contains(_query.toLowerCase())) return const SizedBox();
                return GestureDetector(
                  onTap: ()=> _showPreview(t),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(14),
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Image.network(t['thumbnail'], fit: BoxFit.cover),
                        Container(color: Colors.black26),
                        Positioned(left:8,top:8,child: Chip(backgroundColor: Colors.white70,label: Text(t['category'] ?? '', style: const TextStyle(fontSize:11)))),
                        Positioned(bottom:10,left:8,right:8,child: Text(t['title'] ?? '', style: const TextStyle(color:Colors.white,fontWeight: FontWeight.bold))),
                      ],
                    ),
                  ),
                );
              }
            ),
          )
        ],
      ),
    );
  }

  void _showPreview(dynamic t) {
    showModalBottomSheet(context: context, backgroundColor: Colors.transparent, isScrollControlled:true, builder: (_) {
      return DraggableScrollableSheet(initialChildSize:0.7, builder: (c, sc) {
        return Container(
          decoration: const BoxDecoration(color: Color(0xFF0F0F12), borderRadius: BorderRadius.vertical(top: Radius.circular(16))),
          padding: const EdgeInsets.all(12),
          child: ListView(controller: sc, children: [
            Center(child: Container(width:60,height:6,decoration: BoxDecoration(color: Colors.white12,borderRadius: BorderRadius.circular(8)))),
            const SizedBox(height:10),
            Text(t['title'] ?? '', style: GoogleFonts.poppins(fontSize:18,fontWeight: FontWeight.w700, color: Colors.white)),
            const SizedBox(height:8),
            Container(height:180,decoration: BoxDecoration(borderRadius: BorderRadius.circular(12), image: DecorationImage(image: NetworkImage(t['thumbnail']), fit: BoxFit.cover))),
            const SizedBox(height:12),
            Text('Category: ${t['category']}', style: const TextStyle(color: Colors.white70)),
            const SizedBox(height:12),
            ElevatedButton.icon(onPressed: ()=> _useTemplate(t), icon: const Icon(Icons.check), label: const Text('Use Template'), style: ElevatedButton.styleFrom(backgroundColor: Colors.deepPurple)),
            const SizedBox(height:8),
            ElevatedButton.icon(onPressed: ()=> Navigator.pop(context), icon: const Icon(Icons.close), label: const Text('Close'), style: ElevatedButton.styleFrom(backgroundColor: Colors.white10)),
          ]),
        );
      });
    });
  }
}
