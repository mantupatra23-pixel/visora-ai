import 'dart:math';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/api_service.dart';
class EditorScreen extends StatefulWidget {
  final String videoId;
  const EditorScreen({super.key, this.videoId = ''});
  @override State<EditorScreen> createState() => _EditorScreenState();
}
class _EditorScreenState extends State<EditorScreen> {
  Map<String,dynamic> _selected = {};
  int _credits = 0;
  double _price = 0.0;
  bool _submitting = false;

  final Map<String,Map> defaultPricing = {
    'trim': {'credits':1,'price':5},
    'filters': {'credits':2,'price':10},
    'music': {'credits':2,'price':10},
    'bg_change': {'credits':4,'price':20},
    're_render': {'credits_per_level':2,'price_per_level':10}
  };

  void _toggle(String k) {
    setState(() {
      if (_selected.containsKey(k)) _selected.remove(k); else _selected[k]=true;
      _recalc();
    });
  }
  void _recalc() {
    int c=0; double p=0;
    _selected.forEach((k,v){
      final cfg = defaultPricing[k];
      if (cfg==null) return;
      if (k=='re_render') { final lvl = v is int? v:1; c += (cfg['credits_per_level'] as int)*lvl; p += (cfg['price_per_level'] as num)*lvl; }
      else { c += cfg['credits'] as int; p += (cfg['price'] as num).toDouble(); }
    });
    setState(()=>{ _credits=c, _price=double.parse(p.toStringAsFixed(2)) });
  }

  Future<void> _submit() async {
    if (_selected.isEmpty) { ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Koi edit select nahi'))); return; }
    setState(()=>_submitting=true);
    try {
      // example submit job
      final res = await ApiService.createVideoJob(title: 'Edit ${widget.videoId}', script: 'edit request', options: {'edits':_selected});
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Edit job submitted')));
      Navigator.pop(context);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
    } finally { setState(()=>_submitting=false); }
  }

  @override Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Editor'), backgroundColor: Colors.deepPurple),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            Container(height:180, decoration: BoxDecoration(borderRadius: BorderRadius.circular(12), color: Colors.black12), child: const Center(child: Icon(Icons.play_circle_outline, size:64))),
            const SizedBox(height:12),
            Wrap(spacing:8, children: [
              ChoiceChip(label: const Text('Trim'), selected: _selected.containsKey('trim'), onSelected: (_) => _toggle('trim')),
              ChoiceChip(label: const Text('Filters'), selected: _selected.containsKey('filters'), onSelected: (_) => _toggle('filters')),
              ChoiceChip(label: const Text('Music'), selected: _selected.containsKey('music'), onSelected: (_) => _toggle('music')),
              ChoiceChip(label: const Text('Background'), selected: _selected.containsKey('bg_change'), onSelected: (_) => _toggle('bg_change')),
              ChoiceChip(label: const Text('Re-render'), selected: _selected.containsKey('re_render'), onSelected: (_) {
                if (_selected.containsKey('re_render')) { _selected.remove('re_render'); _recalc(); } else {
                  _selected['re_render'] = 1; _recalc();
                }
              }),
            ]),
            const SizedBox(height:12),
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              Text('Est Credits: $_credits', style: TextStyle(fontWeight: FontWeight.bold)),
              Text('Est Price: ₹$_price', style: TextStyle(fontWeight: FontWeight.bold)),
            ]),
            const Spacer(),
            ElevatedButton.icon(onPressed: _submitting?null:_submit, icon: _submitting?const CircularProgressIndicator(color: Colors.white):const Icon(Icons.send), label: const Text('Confirm & Submit'), style: ElevatedButton.styleFrom(backgroundColor: Colors.deepPurple, minimumSize: const Size.fromHeight(48))),
          ],
        ),
      ),
    );
  }
}
